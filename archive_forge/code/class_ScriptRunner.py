import doctest
import errno
import glob
import logging
import os
import shlex
import sys
import textwrap
from .. import osutils, tests, trace
from ..tests import ui_testing
class ScriptRunner:
    """Run a shell-like script from a test.

    Can be used as:

    from breezy.tests import script

    ...

        def test_bug_nnnnn(self):
            sr = script.ScriptRunner()
            sr.run_script(self, '''
            $ brz init
            $ brz do-this
            # Boom, error
            ''')
    """

    def __init__(self):
        self.output_checker = doctest.OutputChecker()
        self.check_options = doctest.ELLIPSIS

    def run_script(self, test_case, text, null_output_matches_anything=False):
        """Run a shell-like script as a test.

        :param test_case: A TestCase instance that should provide the fail(),
            assertEqualDiff and _run_bzr_core() methods as well as a 'test_dir'
            attribute used as a jail root.

        :param text: A shell-like script (see _script_to_commands for syntax).

        :param null_output_matches_anything: For commands with no specified
            output, ignore any output that does happen, including output on
            standard error.
        """
        self.null_output_matches_anything = null_output_matches_anything
        for cmd, input, output, error in _script_to_commands(text):
            self.run_command(test_case, cmd, input, output, error)

    def run_command(self, test_case, cmd, input, output, error):
        mname = 'do_' + cmd[0]
        method = getattr(self, mname, None)
        if method is None:
            raise SyntaxError('Command not found "{}"'.format(cmd[0]), (None, 1, 1, ' '.join(cmd)))
        if input is None:
            str_input = ''
        else:
            str_input = ''.join(input)
        args = list(self._pre_process_args(cmd[1:]))
        retcode, actual_output, actual_error = method(test_case, str_input, args)
        try:
            self._check_output(output, actual_output, test_case)
        except AssertionError as e:
            raise AssertionError(str(e) + ' in stdout of command %s' % cmd)
        try:
            self._check_output(error, actual_error, test_case)
        except AssertionError as e:
            raise AssertionError(str(e) + ' in stderr of running command %s' % cmd)
        if retcode and (not error) and actual_error:
            test_case.fail('In \n\t%s\nUnexpected error: %s' % (' '.join(cmd), actual_error))
        return (retcode, actual_output, actual_error)

    def _check_output(self, expected, actual, test_case):
        if not actual:
            if expected is None:
                return
            elif expected == '...\n':
                return
            else:
                test_case.fail('expected output: %r, but found nothing' % (expected,))
        null_output_matches_anything = getattr(self, 'null_output_matches_anything', False)
        if null_output_matches_anything and expected is None:
            return
        expected = expected or ''
        matching = self.output_checker.check_output(expected, actual, self.check_options)
        if not matching:
            if expected == actual + '\n':
                pass
            else:
                test_case.assertEqualDiff(expected, actual)

    def _pre_process_args(self, args):
        new_args = []
        for arg in args:
            if arg[0] in ('"', "'") and arg[0] == arg[-1]:
                yield arg[1:-1]
            elif glob.has_magic(arg):
                matches = glob.glob(arg)
                if matches:
                    matches.sort()
                    yield from matches
            else:
                yield arg

    def _read_input(self, input, in_name):
        if in_name is not None:
            infile = open(in_name)
            try:
                input = infile.read()
            finally:
                infile.close()
        return input

    def _write_output(self, output, out_name, out_mode):
        if out_name is not None:
            outfile = open(out_name, out_mode)
            try:
                outfile.write(output)
            finally:
                outfile.close()
            output = None
        return output

    def do_brz(self, test_case, input, args):
        encoding = osutils.get_user_encoding()
        stdout = ui_testing.StringIOWithEncoding()
        stderr = ui_testing.StringIOWithEncoding()
        stdout.encoding = stderr.encoding = encoding
        handler = logging.StreamHandler(stderr)
        handler.setLevel(logging.INFO)
        logger = logging.getLogger('')
        logger.addHandler(handler)
        try:
            retcode = test_case._run_bzr_core(args, encoding=encoding, stdin=input, stdout=stdout, stderr=stderr, working_dir=None)
        finally:
            logger.removeHandler(handler)
        return (retcode, stdout.getvalue(), stderr.getvalue())

    def do_cat(self, test_case, input, args):
        in_name, out_name, out_mode, args = _scan_redirection_options(args)
        if args and in_name is not None:
            raise SyntaxError('Specify a file OR use redirection')
        inputs = []
        if input:
            inputs.append(input)
        input_names = args
        if in_name:
            args.append(in_name)
        for in_name in input_names:
            try:
                inputs.append(self._read_input(None, in_name))
            except OSError as e:
                if e.errno in (errno.ENOENT, errno.EINVAL):
                    return (1, None, '{}: No such file or directory\n'.format(in_name))
                raise
        output = ''.join(inputs)
        try:
            output = self._write_output(output, out_name, out_mode)
        except OSError as e:
            if e.errno in (errno.ENOENT, errno.EINVAL):
                return (1, None, '{}: No such file or directory\n'.format(out_name))
            raise
        return (0, output, None)

    def do_echo(self, test_case, input, args):
        in_name, out_name, out_mode, args = _scan_redirection_options(args)
        if input or in_name:
            raise SyntaxError("echo doesn't read from stdin")
        if args:
            input = ' '.join(args)
        input += '\n'
        output = input
        try:
            output = self._write_output(output, out_name, out_mode)
        except OSError as e:
            if e.errno in (errno.ENOENT, errno.EINVAL):
                return (1, None, '{}: No such file or directory\n'.format(out_name))
            raise
        return (0, output, None)

    def _get_jail_root(self, test_case):
        return test_case.test_dir

    def _ensure_in_jail(self, test_case, path):
        jail_root = self._get_jail_root(test_case)
        if not osutils.is_inside(jail_root, osutils.normalizepath(path)):
            raise ValueError('{} is not inside {}'.format(path, jail_root))

    def do_cd(self, test_case, input, args):
        if len(args) > 1:
            raise SyntaxError('Usage: cd [dir]')
        if len(args) == 1:
            d = args[0]
            self._ensure_in_jail(test_case, d)
        else:
            d = self._get_jail_root(test_case)
        os.chdir(d)
        return (0, None, None)

    def do_mkdir(self, test_case, input, args):
        if not args or len(args) != 1:
            raise SyntaxError('Usage: mkdir dir')
        d = args[0]
        self._ensure_in_jail(test_case, d)
        os.mkdir(d)
        return (0, None, None)

    def do_rm(self, test_case, input, args):
        err = None

        def error(msg, path):
            return "rm: cannot remove '{}': {}\n".format(path, msg)
        force, recursive = (False, False)
        opts = None
        if args and args[0][0] == '-':
            opts = args.pop(0)[1:]
            if 'f' in opts:
                force = True
                opts = opts.replace('f', '', 1)
            if 'r' in opts:
                recursive = True
                opts = opts.replace('r', '', 1)
        if not args or opts:
            raise SyntaxError('Usage: rm [-fr] path+')
        for p in args:
            self._ensure_in_jail(test_case, p)
            try:
                os.remove(p)
            except OSError as e:
                if e.errno in (errno.EISDIR, errno.EPERM, errno.EACCES):
                    if recursive:
                        osutils.rmtree(p)
                    else:
                        err = error('Is a directory', p)
                        break
                elif e.errno == errno.ENOENT:
                    if not force:
                        err = error('No such file or directory', p)
                        break
                else:
                    raise
        if err:
            retcode = 1
        else:
            retcode = 0
        return (retcode, None, err)

    def do_mv(self, test_case, input, args):
        err = None

        def error(msg, src, dst):
            return 'mv: cannot move {} to {}: {}\n'.format(src, dst, msg)
        if not args or len(args) != 2:
            raise SyntaxError('Usage: mv path1 path2')
        src, dst = args
        try:
            real_dst = dst
            if os.path.isdir(dst):
                real_dst = os.path.join(dst, os.path.basename(src))
            os.rename(src, real_dst)
        except OSError as e:
            if e.errno == errno.ENOENT:
                err = error('No such file or directory', src, dst)
            else:
                raise
        if err:
            retcode = 1
        else:
            retcode = 0
        return (retcode, None, err)