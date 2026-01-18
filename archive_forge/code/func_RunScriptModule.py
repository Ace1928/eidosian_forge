from __future__ import print_function
import os
import re
import sys
def RunScriptModule(module):
    """Run a module as a script.

  Locates the module's file and runs it in the current interpreter, or
  optionally a debugger.

  Args:
    module: The module object to run.
  """
    args = sys.argv[1:]
    debug_binary = False
    debugger = 'gdb --args'
    debug_script = False
    show_command_and_exit = False
    while args:
        if args[0] == '--helpstub':
            PrintOurUsage()
            sys.exit(0)
        if args[0] == '--debug_binary':
            debug_binary = True
            args = args[1:]
            continue
        if args[0] == '--debug_script':
            debug_script = True
            args = args[1:]
            continue
        if args[0] == '--show_command_and_exit':
            show_command_and_exit = True
            args = args[1:]
            continue
        matchobj = re.match('--debugger=(.+)', args[0])
        if matchobj is not None:
            debugger = StripQuotes(matchobj.group(1))
            args = args[1:]
            continue
        break
    main_filename = module.__file__
    assert os.path.exists(main_filename), 'Cannot exec() %r: file not found.' % main_filename
    assert os.access(main_filename, os.R_OK), 'Cannot exec() %r: file not readable.' % main_filename
    args = [main_filename] + args
    if debug_binary:
        debugger_args = debugger.split()
        program = debugger_args[0]
        if not os.path.isabs(program):
            program = FindEnv(program)
        python_path = sys.executable
        command_vec = [python_path]
        if debug_script:
            command_vec.extend(GetPdbArgs(python_path))
        args = [program] + debugger_args[1:] + command_vec + args
    elif debug_script:
        args = [sys.executable] + GetPdbArgs(program) + args
    else:
        program = sys.executable
        args = [sys.executable] + args
    if show_command_and_exit:
        print('program: "%s"' % program)
        print('args:', args)
        sys.exit(0)
    try:
        sys.stdout.flush()
        os.execv(program, args)
    except EnvironmentError as e:
        if not getattr(e, 'filename', None):
            e.filename = program
        raise