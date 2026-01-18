from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def AppcommandsUsage(shorthelp=0, writeto_stdout=0, detailed_error=None, exitcode=None, show_cmd=None, show_global_flags=False):
    """Output usage or help information.

  Extracts the __doc__ string from the __main__ module and writes it to
  stderr. If that string contains a '%s' then that is replaced by the command
  pathname. Otherwise a default usage string is being generated.

  The output varies depending on the following:
  - FLAGS.help
  - FLAGS.helpshort
  - show_cmd
  - show_global_flags

  Args:
    shorthelp:      print only command and main module flags, rather than all.
    writeto_stdout: write help message to stdout, rather than to stderr.
    detailed_error: additional details about why usage info was presented.
    exitcode:       if set, exit with this status code after writing help.
    show_cmd:       show help for this command only (name of command).
    show_global_flags: show help for global flags.
  """
    if writeto_stdout:
        stdfile = sys.stdout
    else:
        stdfile = sys.stderr
    prefix = ''.rjust(GetMaxCommandLength() + 2)
    doc = sys.modules['__main__'].__doc__
    if doc:
        help_msg = flags.DocToHelp(doc.replace('%s', sys.argv[0]))
        stdfile.write(flags.TextWrap(help_msg, flags.GetHelpWidth()))
        stdfile.write('\n\n\n')
    if not doc or doc.find('%s') == -1:
        synopsis = 'USAGE: ' + GetSynopsis()
        stdfile.write(flags.TextWrap(synopsis, flags.GetHelpWidth(), '       ', ''))
        stdfile.write('\n\n\n')
    if len(GetCommandList()) == 1:
        cmd_names = []
    else:
        if show_cmd is None or show_cmd == 'help':
            cmd_names = GetCommandList().keys()
            cmd_names.sort()
            stdfile.write('Any of the following commands:\n')
            doc = ', '.join(cmd_names)
            stdfile.write(flags.TextWrap(doc, flags.GetHelpWidth(), '  '))
            stdfile.write('\n\n\n')
        if show_cmd is not None:
            cmd_names = [show_cmd]
        elif FLAGS.help or FLAGS.helpshort or shorthelp:
            cmd_names = []
        else:
            cmd_names = GetCommandList().keys()
            cmd_names.sort()
    for name in cmd_names:
        command = GetCommandByName(name)
        cmd_help = command.CommandGetHelp(GetCommandArgv(), cmd_names=cmd_names)
        cmd_help = cmd_help.strip()
        all_names = ', '.join([name] + (command.CommandGetAliases() or []))
        if len(all_names) + 1 >= len(prefix) or not cmd_help:
            stdfile.write(flags.TextWrap(all_names, flags.GetHelpWidth()))
            stdfile.write('\n')
            prefix1 = prefix
        else:
            prefix1 = all_names.ljust(GetMaxCommandLength() + 2)
        if cmd_help:
            stdfile.write(flags.TextWrap(cmd_help, flags.GetHelpWidth(), prefix, prefix1))
            stdfile.write('\n\n')
        else:
            stdfile.write('\n')
        if len(cmd_names) == 1:
            cmd_flags = command._command_flags
            if cmd_flags.RegisteredFlags():
                stdfile.write('%sFlags for %s:\n' % (prefix, name))
                stdfile.write(cmd_flags.GetHelp(prefix + '  '))
                stdfile.write('\n\n')
    stdfile.write('\n')
    if show_global_flags:
        stdfile.write('Global flags:\n')
        if shorthelp:
            stdfile.write(FLAGS.MainModuleHelp())
        else:
            stdfile.write(FLAGS.GetHelp())
        stdfile.write('\n')
    else:
        stdfile.write("Run '%s --help' to get help for global flags." % GetAppBasename())
    stdfile.write('\n%s\n' % _UsageFooter(detailed_error, cmd_names))
    if exitcode is not None:
        sys.exit(exitcode)