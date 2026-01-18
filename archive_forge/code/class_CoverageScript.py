from __future__ import annotations
import glob
import optparse     # pylint: disable=deprecated-module
import os
import os.path
import shlex
import sys
import textwrap
import traceback
from typing import cast, Any, NoReturn
import coverage
from coverage import Coverage
from coverage import env
from coverage.collector import HAS_CTRACER
from coverage.config import CoverageConfig
from coverage.control import DEFAULT_DATAFILE
from coverage.data import combinable_files, debug_data_file
from coverage.debug import info_header, short_stack, write_formatted_info
from coverage.exceptions import _BaseCoverageException, _ExceptionDuringRun, NoSource
from coverage.execfile import PyRunner
from coverage.results import Numbers, should_fail_under
from coverage.version import __url__
class CoverageScript:
    """The command-line interface to coverage.py."""

    def __init__(self) -> None:
        self.global_option = False
        self.coverage: Coverage

    def command_line(self, argv: list[str]) -> int:
        """The bulk of the command line interface to coverage.py.

        `argv` is the argument list to process.

        Returns 0 if all is well, 1 if something went wrong.

        """
        if not argv:
            show_help(topic='minimum_help')
            return OK
        parser: optparse.OptionParser | None
        self.global_option = argv[0].startswith('-')
        if self.global_option:
            parser = GlobalOptionParser()
        else:
            parser = COMMANDS.get(argv[0])
            if not parser:
                show_help(f'Unknown command: {argv[0]!r}')
                return ERR
            argv = argv[1:]
        ok, options, args = parser.parse_args_ok(argv)
        if not ok:
            return ERR
        assert options is not None
        if self.do_help(options, args, parser):
            return OK
        source = unshell_list(options.source)
        omit = unshell_list(options.omit)
        include = unshell_list(options.include)
        debug = unshell_list(options.debug)
        contexts = unshell_list(options.contexts)
        if options.concurrency is not None:
            concurrency = options.concurrency.split(',')
        else:
            concurrency = None
        self.coverage = Coverage(data_file=options.data_file or DEFAULT_DATAFILE, data_suffix=options.parallel_mode, cover_pylib=options.pylib, timid=options.timid, branch=options.branch, config_file=options.rcfile, source=source, omit=omit, include=include, debug=debug, concurrency=concurrency, check_preimported=True, context=options.context, messages=not options.quiet)
        if options.action == 'debug':
            return self.do_debug(args)
        elif options.action == 'erase':
            self.coverage.erase()
            return OK
        elif options.action == 'run':
            return self.do_run(options, args)
        elif options.action == 'combine':
            if options.append:
                self.coverage.load()
            data_paths = args or None
            self.coverage.combine(data_paths, strict=True, keep=bool(options.keep))
            self.coverage.save()
            return OK
        report_args = dict(morfs=unglob_args(args), ignore_errors=options.ignore_errors, omit=omit, include=include, contexts=contexts)
        sys.path.insert(0, '')
        self.coverage.load()
        total = None
        if options.action == 'report':
            total = self.coverage.report(precision=options.precision, show_missing=options.show_missing, skip_covered=options.skip_covered, skip_empty=options.skip_empty, sort=options.sort, output_format=options.format, **report_args)
        elif options.action == 'annotate':
            self.coverage.annotate(directory=options.directory, **report_args)
        elif options.action == 'html':
            total = self.coverage.html_report(directory=options.directory, precision=options.precision, skip_covered=options.skip_covered, skip_empty=options.skip_empty, show_contexts=options.show_contexts, title=options.title, **report_args)
        elif options.action == 'xml':
            total = self.coverage.xml_report(outfile=options.outfile, skip_empty=options.skip_empty, **report_args)
        elif options.action == 'json':
            total = self.coverage.json_report(outfile=options.outfile, pretty_print=options.pretty_print, show_contexts=options.show_contexts, **report_args)
        elif options.action == 'lcov':
            total = self.coverage.lcov_report(outfile=options.outfile, **report_args)
        else:
            raise AssertionError
        if total is not None:
            if options.fail_under is not None:
                self.coverage.set_option('report:fail_under', options.fail_under)
            if options.precision is not None:
                self.coverage.set_option('report:precision', options.precision)
            fail_under = cast(float, self.coverage.get_option('report:fail_under'))
            precision = cast(int, self.coverage.get_option('report:precision'))
            if should_fail_under(total, fail_under, precision):
                msg = 'total of {total} is less than fail-under={fail_under:.{p}f}'.format(total=Numbers(precision=precision).display_covered(total), fail_under=fail_under, p=precision)
                print('Coverage failure:', msg)
                return FAIL_UNDER
        return OK

    def do_help(self, options: optparse.Values, args: list[str], parser: optparse.OptionParser) -> bool:
        """Deal with help requests.

        Return True if it handled the request, False if not.

        """
        if options.help:
            if self.global_option:
                show_help(topic='help')
            else:
                show_help(parser=parser)
            return True
        if options.action == 'help':
            if args:
                for a in args:
                    parser_maybe = COMMANDS.get(a)
                    if parser_maybe is not None:
                        show_help(parser=parser_maybe)
                    else:
                        show_help(topic=a)
            else:
                show_help(topic='help')
            return True
        if options.version:
            show_help(topic='version')
            return True
        return False

    def do_run(self, options: optparse.Values, args: list[str]) -> int:
        """Implementation of 'coverage run'."""
        if not args:
            if options.module:
                show_help('No module specified for -m')
                return ERR
            command_line = cast(str, self.coverage.get_option('run:command_line'))
            if command_line is not None:
                args = shlex.split(command_line)
                if args and args[0] in {'-m', '--module'}:
                    options.module = True
                    args = args[1:]
        if not args:
            show_help('Nothing to do.')
            return ERR
        if options.append and self.coverage.get_option('run:parallel'):
            show_help("Can't append to data files in parallel mode.")
            return ERR
        if options.concurrency == 'multiprocessing':
            for opt_name in ['branch', 'include', 'omit', 'pylib', 'source', 'timid']:
                if getattr(options, opt_name) is not None:
                    show_help('Options affecting multiprocessing must only be specified ' + 'in a configuration file.\n' + f'Remove --{opt_name} from the command line.')
                    return ERR
        os.environ['COVERAGE_RUN'] = 'true'
        runner = PyRunner(args, as_module=bool(options.module))
        runner.prepare()
        if options.append:
            self.coverage.load()
        self.coverage.start()
        code_ran = True
        try:
            runner.run()
        except NoSource:
            code_ran = False
            raise
        finally:
            self.coverage.stop()
            if code_ran:
                self.coverage.save()
        return OK

    def do_debug(self, args: list[str]) -> int:
        """Implementation of 'coverage debug'."""
        if not args:
            show_help('What information would you like: config, data, sys, premain, pybehave?')
            return ERR
        if args[1:]:
            show_help('Only one topic at a time, please')
            return ERR
        if args[0] == 'sys':
            write_formatted_info(print, 'sys', self.coverage.sys_info())
        elif args[0] == 'data':
            print(info_header('data'))
            data_file = self.coverage.config.data_file
            debug_data_file(data_file)
            for filename in combinable_files(data_file):
                print('-----')
                debug_data_file(filename)
        elif args[0] == 'config':
            write_formatted_info(print, 'config', self.coverage.config.debug_info())
        elif args[0] == 'premain':
            print(info_header('premain'))
            print(short_stack(full=True))
        elif args[0] == 'pybehave':
            write_formatted_info(print, 'pybehave', env.debug_info())
        else:
            show_help(f"Don't know what you mean by {args[0]!r}")
            return ERR
        return OK