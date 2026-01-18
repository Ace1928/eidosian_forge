from __future__ import absolute_import
import os
from .. import Utils
class CompilationOptions(object):
    """
    See default_options at the end of this module for a list of all possible
    options and CmdLine.usage and CmdLine.parse_command_line() for their
    meaning.
    """

    def __init__(self, defaults=None, **kw):
        self.include_path = []
        if defaults:
            if isinstance(defaults, CompilationOptions):
                defaults = defaults.__dict__
        else:
            defaults = default_options
        options = dict(defaults)
        options.update(kw)
        unknown_options = set(options) - set(default_options)
        unknown_options.difference_update(['include_path'])
        if unknown_options:
            message = 'got unknown compilation option%s, please remove: %s' % ('s' if len(unknown_options) > 1 else '', ', '.join(unknown_options))
            raise ValueError(message)
        directive_defaults = get_directive_defaults()
        directives = dict(options['compiler_directives'])
        unknown_directives = set(directives) - set(directive_defaults)
        if unknown_directives:
            message = 'got unknown compiler directive%s: %s' % ('s' if len(unknown_directives) > 1 else '', ', '.join(unknown_directives))
            raise ValueError(message)
        options['compiler_directives'] = directives
        if directives.get('np_pythran', False) and (not options['cplus']):
            import warnings
            warnings.warn('C++ mode forced when in Pythran mode!')
            options['cplus'] = True
        if 'language_level' not in kw and directives.get('language_level'):
            options['language_level'] = directives['language_level']
        elif not options.get('language_level'):
            options['language_level'] = directive_defaults.get('language_level')
        if 'formal_grammar' in directives and 'formal_grammar' not in kw:
            options['formal_grammar'] = directives['formal_grammar']
        if options['cache'] is True:
            options['cache'] = os.path.join(Utils.get_cython_cache_dir(), 'compiler')
        self.__dict__.update(options)

    def configure_language_defaults(self, source_extension):
        if source_extension == 'py':
            if self.compiler_directives.get('binding') is None:
                self.compiler_directives['binding'] = True

    def get_fingerprint(self):
        """
        Return a string that contains all the options that are relevant for cache invalidation.
        """
        data = {}
        for key, value in self.__dict__.items():
            if key in ['show_version', 'errors_to_stderr', 'verbose', 'quiet']:
                continue
            elif key in ['output_file', 'output_dir']:
                continue
            elif key in ['depfile']:
                continue
            elif key in ['timestamps']:
                continue
            elif key in ['cache']:
                continue
            elif key in ['compiler_directives']:
                continue
            elif key in ['include_path']:
                continue
            elif key in ['working_path']:
                continue
            elif key in ['create_extension']:
                continue
            elif key in ['build_dir']:
                continue
            elif key in ['use_listing_file', 'generate_pxi', 'annotate', 'annotate_coverage_xml']:
                data[key] = value
            elif key in ['formal_grammar', 'evaluate_tree_assertions']:
                data[key] = value
            elif key in ['embedded_metadata', 'emit_linenums', 'c_line_in_traceback', 'gdb_debug', 'relative_path_in_code_position_comments']:
                data[key] = value
            elif key in ['cplus', 'language_level', 'compile_time_env', 'np_pythran']:
                data[key] = value
            elif key == ['capi_reexport_cincludes']:
                if self.capi_reexport_cincludes:
                    raise NotImplementedError('capi_reexport_cincludes is not compatible with Cython caching')
            elif key == ['common_utility_include_dir']:
                if self.common_utility_include_dir:
                    raise NotImplementedError('common_utility_include_dir is not compatible with Cython caching yet')
            else:
                data[key] = value

        def to_fingerprint(item):
            """
            Recursively turn item into a string, turning dicts into lists with
            deterministic ordering.
            """
            if isinstance(item, dict):
                item = sorted([(repr(key), to_fingerprint(value)) for key, value in item.items()])
            return repr(item)
        return to_fingerprint(data)