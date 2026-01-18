from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def _CommandsStart():
    """Main initialization.

  This initializes flag values, and calls __main__.main().  Only non-flag
  arguments are passed to main().  The return value of main() is used as the
  exit status.

  """
    app.RegisterAndParseFlagsWithUsage()
    try:
        sys.modules['__main__'].main(GetCommandArgv())
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as error:
        traceback.print_exc()
        ShortHelpAndExit('\nFATAL error in main: %s' % error)
    if len(GetCommandArgv()) > 1:
        command = GetCommand(command_required=True)
    else:
        command = GetCommandByName('help')
    sys.exit(command.CommandRun(GetCommandArgv()))