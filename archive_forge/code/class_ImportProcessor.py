import sys
import time
from . import errors
from .helpers import newobject as object
class ImportProcessor(object):
    """Base class for fast-import stream processors.

    Subclasses should override the pre_*, post_* and *_handler
    methods as appropriate.
    """
    known_params = []

    def __init__(self, params=None, verbose=False, outf=None):
        if outf is None:
            self.outf = sys.stdout
        else:
            self.outf = outf
        self.verbose = verbose
        if params is None:
            self.params = {}
        else:
            self.params = params
            self.validate_parameters()
        self.finished = False

    def validate_parameters(self):
        """Validate that the parameters are correctly specified."""
        for p in self.params:
            if p not in self.known_params:
                raise errors.UnknownParameter(p, self.known_params)

    def process(self, command_iter):
        """Import data into Bazaar by processing a stream of commands.

        :param command_iter: an iterator providing commands
        """
        self._process(command_iter)

    def _process(self, command_iter):
        self.pre_process()
        for cmd in command_iter():
            try:
                name = (cmd.name + b'_handler').decode('utf8')
                handler = getattr(self.__class__, name)
            except KeyError:
                raise errors.MissingHandler(cmd.name)
            else:
                self.pre_handler(cmd)
                handler(self, cmd)
                self.post_handler(cmd)
            if self.finished:
                break
        self.post_process()

    def warning(self, msg, *args):
        """Output a warning but timestamp it."""
        pass

    def debug(self, mgs, *args):
        """Output a debug message."""
        pass

    def _time_of_day(self):
        """Time of day as a string."""
        return time.strftime('%H:%M:%S')

    def pre_process(self):
        """Hook for logic at start of processing."""
        pass

    def post_process(self):
        """Hook for logic at end of processing."""
        pass

    def pre_handler(self, cmd):
        """Hook for logic before each handler starts."""
        pass

    def post_handler(self, cmd):
        """Hook for logic after each handler finishes."""
        pass

    def progress_handler(self, cmd):
        """Process a ProgressCommand."""
        raise NotImplementedError(self.progress_handler)

    def blob_handler(self, cmd):
        """Process a BlobCommand."""
        raise NotImplementedError(self.blob_handler)

    def checkpoint_handler(self, cmd):
        """Process a CheckpointCommand."""
        raise NotImplementedError(self.checkpoint_handler)

    def commit_handler(self, cmd):
        """Process a CommitCommand."""
        raise NotImplementedError(self.commit_handler)

    def reset_handler(self, cmd):
        """Process a ResetCommand."""
        raise NotImplementedError(self.reset_handler)

    def tag_handler(self, cmd):
        """Process a TagCommand."""
        raise NotImplementedError(self.tag_handler)

    def feature_handler(self, cmd):
        """Process a FeatureCommand."""
        raise NotImplementedError(self.feature_handler)