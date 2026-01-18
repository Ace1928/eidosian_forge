from abc import ABCMeta
from abc import abstractmethod
class TBLoader:
    """TBPlugin factory base class.

    Plugins can override this class to customize how a plugin is loaded at
    startup. This might entail adding command-line arguments, checking if
    optional dependencies are installed, and potentially also specializing
    the plugin class at runtime.

    When plugins use optional dependencies, the loader needs to be
    specified in its own module. That way it's guaranteed to be
    importable, even if the `TBPlugin` itself can't be imported.

    Subclasses must have trivial constructors.
    """

    def define_flags(self, parser):
        """Adds plugin-specific CLI flags to parser.

        The default behavior is to do nothing.

        When overriding this method, it's recommended that plugins call the
        `parser.add_argument_group(plugin_name)` method for readability. No
        flags should be specified that would cause `parse_args([])` to fail.

        Args:
          parser: The argument parsing object, which may be mutated.
        """
        pass

    def fix_flags(self, flags):
        """Allows flag values to be corrected or validated after parsing.

        Args:
          flags: The parsed argparse.Namespace object.

        Raises:
          base_plugin.FlagsError: If a flag is invalid or a required
              flag is not passed.
        """
        pass

    def load(self, context):
        """Loads a TBPlugin instance during the setup phase.

        Args:
          context: The TBContext instance.

        Returns:
          A plugin instance or None if it could not be loaded. Loaders that return
          None are skipped.

        :type context: TBContext
        :rtype: TBPlugin | None
        """
        return None