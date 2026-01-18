import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
class ASTCodeUpgrader:
    """Handles upgrading a set of Python files using a given API change spec."""

    def __init__(self, api_change_spec):
        if not isinstance(api_change_spec, APIChangeSpec):
            raise TypeError('Must pass APIChangeSpec to ASTCodeUpgrader, got %s' % type(api_change_spec))
        self._api_change_spec = api_change_spec

    def process_file(self, in_filename, out_filename, no_change_to_outfile_on_error=False):
        """Process the given python file for incompatible changes.

    Args:
      in_filename: filename to parse
      out_filename: output file to write to
      no_change_to_outfile_on_error: not modify the output file on errors
    Returns:
      A tuple representing number of files processed, log of actions, errors
    """
        with open(in_filename, 'r') as in_file, tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
            ret = self.process_opened_file(in_filename, in_file, out_filename, temp_file)
        if no_change_to_outfile_on_error and ret[0] == 0:
            os.remove(temp_file.name)
        else:
            shutil.move(temp_file.name, out_filename)
        return ret

    def format_log(self, log, in_filename):
        log_string = '%d:%d: %s: %s' % (log[1], log[2], log[0], log[3])
        if in_filename:
            return in_filename + ':' + log_string
        else:
            return log_string

    def update_string_pasta(self, text, in_filename):
        """Updates a file using pasta."""
        try:
            t = pasta.parse(text)
        except (SyntaxError, ValueError, TypeError):
            log = ['ERROR: Failed to parse.\n' + traceback.format_exc()]
            return (0, '', log, [])
        t, preprocess_logs, preprocess_errors = self._api_change_spec.preprocess(t)
        visitor = _PastaEditVisitor(self._api_change_spec)
        visitor.visit(t)
        self._api_change_spec.clear_preprocessing()
        logs = [self.format_log(log, None) for log in preprocess_logs + visitor.log]
        errors = [self.format_log(error, in_filename) for error in preprocess_errors + visitor.warnings_and_errors]
        return (1, pasta.dump(t), logs, errors)

    def _format_log(self, log, in_filename, out_filename):
        text = '-' * 80 + '\n'
        text += 'Processing file %r\n outputting to %r\n' % (in_filename, out_filename)
        text += '-' * 80 + '\n\n'
        text += '\n'.join(log) + '\n'
        text += '-' * 80 + '\n\n'
        return text

    def process_opened_file(self, in_filename, in_file, out_filename, out_file):
        """Process the given python file for incompatible changes.

    This function is split out to facilitate StringIO testing from
    tf_upgrade_test.py.

    Args:
      in_filename: filename to parse
      in_file: opened file (or StringIO)
      out_filename: output file to write to
      out_file: opened file (or StringIO)
    Returns:
      A tuple representing number of files processed, log of actions, errors
    """
        lines = in_file.readlines()
        processed_file, new_file_content, log, process_errors = self.update_string_pasta(''.join(lines), in_filename)
        if out_file and processed_file:
            out_file.write(new_file_content)
        return (processed_file, self._format_log(log, in_filename, out_filename), process_errors)

    def process_tree(self, root_directory, output_root_directory, copy_other_files):
        """Processes upgrades on an entire tree of python files in place.

    Note that only Python files. If you have custom code in other languages,
    you will need to manually upgrade those.

    Args:
      root_directory: Directory to walk and process.
      output_root_directory: Directory to use as base.
      copy_other_files: Copy files that are not touched by this converter.

    Returns:
      A tuple of files processed, the report string for all files, and a dict
        mapping filenames to errors encountered in that file.
    """
        if output_root_directory == root_directory:
            return self.process_tree_inplace(root_directory)
        if output_root_directory and os.path.exists(output_root_directory):
            print('Output directory %r must not already exist.' % output_root_directory)
            sys.exit(1)
        norm_root = os.path.split(os.path.normpath(root_directory))
        norm_output = os.path.split(os.path.normpath(output_root_directory))
        if norm_root == norm_output:
            print('Output directory %r same as input directory %r' % (root_directory, output_root_directory))
            sys.exit(1)
        files_to_process = []
        files_to_copy = []
        for dir_name, _, file_list in os.walk(root_directory):
            py_files = [f for f in file_list if f.endswith('.py')]
            copy_files = [f for f in file_list if not f.endswith('.py')]
            for filename in py_files:
                fullpath = os.path.join(dir_name, filename)
                fullpath_output = os.path.join(output_root_directory, os.path.relpath(fullpath, root_directory))
                files_to_process.append((fullpath, fullpath_output))
            if copy_other_files:
                for filename in copy_files:
                    fullpath = os.path.join(dir_name, filename)
                    fullpath_output = os.path.join(output_root_directory, os.path.relpath(fullpath, root_directory))
                    files_to_copy.append((fullpath, fullpath_output))
        file_count = 0
        tree_errors = {}
        report = ''
        report += '=' * 80 + '\n'
        report += 'Input tree: %r\n' % root_directory
        report += '=' * 80 + '\n'
        for input_path, output_path in files_to_process:
            output_directory = os.path.dirname(output_path)
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)
            if os.path.islink(input_path):
                link_target = os.readlink(input_path)
                link_target_output = os.path.join(output_root_directory, os.path.relpath(link_target, root_directory))
                if (link_target, link_target_output) in files_to_process:
                    os.symlink(link_target_output, output_path)
                else:
                    report += 'Copying symlink %s without modifying its target %s' % (input_path, link_target)
                    os.symlink(link_target, output_path)
                continue
            file_count += 1
            _, l_report, l_errors = self.process_file(input_path, output_path)
            tree_errors[input_path] = l_errors
            report += l_report
        for input_path, output_path in files_to_copy:
            output_directory = os.path.dirname(output_path)
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)
            shutil.copy(input_path, output_path)
        return (file_count, report, tree_errors)

    def process_tree_inplace(self, root_directory):
        """Process a directory of python files in place."""
        files_to_process = []
        for dir_name, _, file_list in os.walk(root_directory):
            py_files = [os.path.join(dir_name, f) for f in file_list if f.endswith('.py')]
            files_to_process += py_files
        file_count = 0
        tree_errors = {}
        report = ''
        report += '=' * 80 + '\n'
        report += 'Input tree: %r\n' % root_directory
        report += '=' * 80 + '\n'
        for path in files_to_process:
            if os.path.islink(path):
                report += 'Skipping symlink %s.\n' % path
                continue
            file_count += 1
            _, l_report, l_errors = self.process_file(path, path)
            tree_errors[path] = l_errors
            report += l_report
        return (file_count, report, tree_errors)