from __future__ import print_function
from .. import (
class QueryProcessor(processor.ImportProcessor):
    """An import processor that queries the input.

    No changes to the current repository are made.
    """
    known_params = commands.COMMAND_NAMES + commands.FILE_COMMAND_NAMES + [b'commit-mark']

    def __init__(self, params=None, verbose=False):
        processor.ImportProcessor.__init__(self, params, verbose)
        self.parsed_params = {}
        self.interesting_commit = None
        self._finished = False
        if params:
            if 'commit-mark' in params:
                self.interesting_commit = params['commit-mark']
                del params['commit-mark']
            for name, value in params.items():
                if value == 1:
                    fields = None
                else:
                    fields = value.split(',')
                self.parsed_params[name] = fields

    def pre_handler(self, cmd):
        """Hook for logic before each handler starts."""
        if self._finished:
            return
        if self.interesting_commit and cmd.name == 'commit':
            if cmd.mark == self.interesting_commit:
                print(cmd.to_string())
                self._finished = True
            return
        if cmd.name in self.parsed_params:
            fields = self.parsed_params[cmd.name]
            str = cmd.dump_str(fields, self.parsed_params, self.verbose)
            print('%s' % (str,))

    def progress_handler(self, cmd):
        """Process a ProgressCommand."""
        pass

    def blob_handler(self, cmd):
        """Process a BlobCommand."""
        pass

    def checkpoint_handler(self, cmd):
        """Process a CheckpointCommand."""
        pass

    def commit_handler(self, cmd):
        """Process a CommitCommand."""
        pass

    def reset_handler(self, cmd):
        """Process a ResetCommand."""
        pass

    def tag_handler(self, cmd):
        """Process a TagCommand."""
        pass

    def feature_handler(self, cmd):
        """Process a FeatureCommand."""
        feature = cmd.feature_name
        if feature not in commands.FEATURE_NAMES:
            self.warning('feature %s is not supported - parsing may fail' % (feature,))