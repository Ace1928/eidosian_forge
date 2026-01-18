import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
class cmd_bisect(Command):
    """Find an interesting commit using a binary search.

    Bisecting, in a nutshell, is a way to find the commit at which
    some testable change was made, such as the introduction of a bug
    or feature.  By identifying a version which did not have the
    interesting change and a later version which did, a developer
    can test for the presence of the change at various points in
    the history, eventually ending up at the precise commit when
    the change was first introduced.

    This command uses subcommands to implement the search, each
    of which changes the state of the bisection.  The
    subcommands are:

    brz bisect start
        Start a bisect, possibly clearing out a previous bisect.

    brz bisect yes [-r rev]
        The specified revision (or the current revision, if not given)
        has the characteristic we're looking for,

    brz bisect no [-r rev]
        The specified revision (or the current revision, if not given)
        does not have the characteristic we're looking for,

    brz bisect move -r rev
        Switch to a different revision manually.  Use if the bisect
        algorithm chooses a revision that is not suitable.  Try to
        move as little as possible.

    brz bisect reset
        Clear out a bisection in progress.

    brz bisect log [-o file]
        Output a log of the current bisection to standard output, or
        to the specified file.

    brz bisect replay <logfile>
        Replay a previously-saved bisect log, forgetting any bisection
        that might be in progress.

    brz bisect run <script>
        Bisect automatically using <script> to determine 'yes' or 'no'.
        <script> should exit with:
           0 for yes
           125 for unknown (like build failed so we could not test)
           anything else for no
    """
    takes_args = ['subcommand', 'args*']
    takes_options = [Option('output', short_name='o', help='Write log to this file.', type=str), 'revision', 'directory']

    def _check(self, controldir):
        """Check preconditions for most operations to work."""
        if not controldir.control_transport.has(BISECT_INFO_PATH):
            raise CommandError('No bisection in progress.')

    def _set_state(self, controldir, revspec, state):
        """Set the state of the given revspec and bisecting.

        Returns boolean indicating if bisection is done."""
        bisect_log = BisectLog(controldir)
        if bisect_log.is_done():
            note('No further bisection is possible.\n')
            bisect_log._current.show_rev_log(outf=self.outf)
            return True
        if revspec:
            bisect_log.set_status_from_revspec(revspec, state)
        else:
            bisect_log.set_current(state)
        bisect_log.bisect(self.outf)
        bisect_log.save()
        return False

    def run(self, subcommand, args_list, directory='.', revision=None, output=None):
        """Handle the bisect command."""
        log_fn = None
        if subcommand in ('yes', 'no', 'move') and revision:
            pass
        elif subcommand in ('replay',) and args_list and (len(args_list) == 1):
            log_fn = args_list[0]
        elif subcommand in ('move',) and (not revision):
            raise CommandError("The 'bisect move' command requires a revision.")
        elif subcommand in ('run',):
            run_script = args_list[0]
        elif args_list or revision:
            raise CommandError('Improper arguments to bisect ' + subcommand)
        controldir, _ = ControlDir.open_containing(directory)
        if subcommand == 'start':
            self.start(controldir)
        elif subcommand == 'yes':
            self.yes(controldir, revision)
        elif subcommand == 'no':
            self.no(controldir, revision)
        elif subcommand == 'move':
            self.move(controldir, revision)
        elif subcommand == 'reset':
            self.reset(controldir)
        elif subcommand == 'log':
            self.log(controldir, output)
        elif subcommand == 'replay':
            self.replay(controldir, log_fn)
        elif subcommand == 'run':
            self.run_bisect(controldir, run_script)
        else:
            raise CommandError('Unknown bisect command: ' + subcommand)

    def reset(self, controldir):
        """Reset the bisect state to no state."""
        self._check(controldir)
        BisectCurrent(controldir).reset()
        controldir.control_transport.delete(BISECT_INFO_PATH)

    def start(self, controldir):
        """Reset the bisect state, then prepare for a new bisection."""
        if controldir.control_transport.has(BISECT_INFO_PATH):
            BisectCurrent(controldir).reset()
            controldir.control_transport.delete(BISECT_INFO_PATH)
        bisect_log = BisectLog(controldir)
        bisect_log.set_current('start')
        bisect_log.save()

    def yes(self, controldir, revspec):
        """Mark that a given revision has the state we're looking for."""
        self._set_state(controldir, revspec, 'yes')

    def no(self, controldir, revspec):
        """Mark a given revision as wrong."""
        self._set_state(controldir, revspec, 'no')

    def move(self, controldir, revspec):
        """Move to a different revision manually."""
        current = BisectCurrent(controldir)
        current.switch(revspec)
        current.show_rev_log(outf=self.outf)

    def log(self, controldir, filename):
        """Write the current bisect log to a file."""
        self._check(controldir)
        bisect_log = BisectLog(controldir)
        bisect_log.change_file_name(filename)
        bisect_log.save()

    def replay(self, controldir, filename):
        """Apply the given log file to a clean state, so the state is
        exactly as it was when the log was saved."""
        if controldir.control_transport.has(BISECT_INFO_PATH):
            BisectCurrent(controldir).reset()
            controldir.control_transport.delete(BISECT_INFO_PATH)
        bisect_log = BisectLog(controldir, filename)
        bisect_log.change_file_name(BISECT_INFO_PATH)
        bisect_log.save()
        bisect_log.bisect(self.outf)

    def run_bisect(self, controldir, script):
        import subprocess
        note('Starting bisect.')
        self.start(controldir)
        while True:
            try:
                process = subprocess.Popen(script, shell=True)
                process.wait()
                retcode = process.returncode
                if retcode == 0:
                    done = self._set_state(controldir, None, 'yes')
                elif retcode == 125:
                    break
                else:
                    done = self._set_state(controldir, None, 'no')
                if done:
                    break
            except RuntimeError:
                break