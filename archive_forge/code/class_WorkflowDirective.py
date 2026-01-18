import sys
import os
import shutil
import io
import re
import textwrap
from os.path import relpath
from errno import EEXIST
import traceback
class WorkflowDirective(Directive):
    has_content = True
    required_arguments = 0
    optional_arguments = 2
    final_argument_whitespace = False
    option_spec = {'alt': directives.unchanged, 'height': directives.length_or_unitless, 'width': directives.length_or_percentage_or_unitless, 'scale': directives.nonnegative_int, 'align': _option_align, 'class': directives.class_option, 'include-source': _option_boolean, 'format': _option_format, 'context': _option_context, 'nofigs': directives.flag, 'encoding': directives.encoding, 'graph2use': _option_graph2use, 'simple_form': _option_boolean}

    def run(self):
        if missing_imports:
            raise ImportError('\n'.join(missing_imports))
        document = self.state_machine.document
        config = document.settings.env.config
        nofigs = 'nofigs' in self.options
        formats = get_wf_formats(config)
        default_fmt = formats[0][0]
        graph2use = self.options.get('graph2use', 'hierarchical')
        simple_form = self.options.get('simple_form', True)
        self.options.setdefault('include-source', config.wf_include_source)
        keep_context = 'context' in self.options
        context_opt = None if not keep_context else self.options['context']
        rst_file = document.attributes['source']
        rst_dir = os.path.dirname(rst_file)
        if len(self.arguments):
            if not config.wf_basedir:
                source_file_name = os.path.join(setup.app.builder.srcdir, directives.uri(self.arguments[0]))
            else:
                source_file_name = os.path.join(setup.confdir, config.wf_basedir, directives.uri(self.arguments[0]))
            caption = '\n'.join(self.content)
            if len(self.arguments) == 2:
                function_name = self.arguments[1]
            else:
                function_name = None
            with io.open(source_file_name, 'r', encoding='utf-8') as fd:
                code = fd.read()
            output_base = os.path.basename(source_file_name)
        else:
            source_file_name = rst_file
            code = textwrap.dedent('\n'.join([str(c) for c in self.content]))
            counter = document.attributes.get('_wf_counter', 0) + 1
            document.attributes['_wf_counter'] = counter
            base, _ = os.path.splitext(os.path.basename(source_file_name))
            output_base = '%s-%d.py' % (base, counter)
            function_name = None
            caption = ''
        base, source_ext = os.path.splitext(output_base)
        if source_ext in ('.py', '.rst', '.txt'):
            output_base = base
        else:
            source_ext = ''
        output_base = output_base.replace('.', '-')
        is_doctest = contains_doctest(code)
        if 'format' in self.options:
            if self.options['format'] == 'python':
                is_doctest = False
            else:
                is_doctest = True
        source_rel_name = relpath(source_file_name, setup.confdir)
        source_rel_dir = os.path.dirname(source_rel_name)
        while source_rel_dir.startswith(os.path.sep):
            source_rel_dir = source_rel_dir[1:]
        build_dir = os.path.join(os.path.dirname(setup.app.doctreedir), 'wf_directive', source_rel_dir)
        build_dir = os.path.normpath(build_dir)
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        dest_dir = os.path.abspath(os.path.join(setup.app.builder.outdir, source_rel_dir))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        dest_dir_link = os.path.join(relpath(setup.confdir, rst_dir), source_rel_dir).replace(os.path.sep, '/')
        try:
            build_dir_link = relpath(build_dir, rst_dir).replace(os.path.sep, '/')
        except ValueError:
            build_dir_link = build_dir
        source_link = dest_dir_link + '/' + output_base + source_ext
        try:
            results = render_figures(code, source_file_name, build_dir, output_base, keep_context, function_name, config, graph2use, simple_form, context_reset=context_opt == 'reset', close_figs=context_opt == 'close-figs')
            errors = []
        except GraphError as err:
            reporter = self.state.memo.reporter
            sm = reporter.system_message(2, 'Exception occurred in plotting %s\n from %s:\n%s' % (output_base, source_file_name, err), line=self.lineno)
            results = [(code, [])]
            errors = [sm]
        caption = '\n'.join(('      ' + line.strip() for line in caption.split('\n')))
        total_lines = []
        for j, (code_piece, images) in enumerate(results):
            if self.options['include-source']:
                if is_doctest:
                    lines = ['']
                    lines += [row.rstrip() for row in code_piece.split('\n')]
                else:
                    lines = ['.. code-block:: python', '']
                    lines += ['    %s' % row.rstrip() for row in code_piece.split('\n')]
                source_code = '\n'.join(lines)
            else:
                source_code = ''
            if nofigs:
                images = []
            opts = [':%s: %s' % (key, val) for key, val in list(self.options.items()) if key in ('alt', 'height', 'width', 'scale', 'align', 'class')]
            only_html = '.. only:: html'
            only_latex = '.. only:: latex'
            only_texinfo = '.. only:: texinfo'
            if j == 0 and config.wf_html_show_source_link:
                src_link = source_link
            else:
                src_link = None
            result = format_template(config.wf_template or TEMPLATE, default_fmt=default_fmt, dest_dir=dest_dir_link, build_dir=build_dir_link, source_link=src_link, multi_image=len(images) > 1, only_html=only_html, only_latex=only_latex, only_texinfo=only_texinfo, options=opts, images=images, source_code=source_code, html_show_formats=config.wf_html_show_formats and len(images), caption=caption)
            total_lines.extend(result.split('\n'))
            total_lines.extend('\n')
        if total_lines:
            self.state_machine.insert_input(total_lines, source=source_file_name)
        os.makedirs(dest_dir, exist_ok=True)
        for code_piece, images in results:
            for img in images:
                for fn in img.filenames():
                    destimg = os.path.join(dest_dir, os.path.basename(fn))
                    if fn != destimg:
                        shutil.copyfile(fn, destimg)
        target_name = os.path.join(dest_dir, output_base + source_ext)
        with io.open(target_name, 'w', encoding='utf-8') as f:
            if source_file_name == rst_file:
                code_escaped = unescape_doctest(code)
            else:
                code_escaped = code
            f.write(code_escaped)
        return errors