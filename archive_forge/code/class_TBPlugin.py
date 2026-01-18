from abc import ABCMeta
from abc import abstractmethod
class TBPlugin(metaclass=ABCMeta):
    """TensorBoard plugin interface.

    Every plugin must extend from this class.

    Subclasses should have a trivial constructor that takes a TBContext
    argument. Any operation that might throw an exception should either be
    done lazily or made safe with a TBLoader subclass, so the plugin won't
    negatively impact the rest of TensorBoard.

    Fields:
      plugin_name: The plugin_name will also be a prefix in the http
        handlers, e.g. `data/plugins/$PLUGIN_NAME/$HANDLER` The plugin
        name must be unique for each registered plugin, or a ValueError
        will be thrown when the application is constructed. The plugin
        name must only contain characters among [A-Za-z0-9_.-], and must
        be nonempty, or a ValueError will similarly be thrown.
    """
    plugin_name = None

    def __init__(self, context):
        """Initializes this plugin.

        The default implementation does nothing. Subclasses are encouraged
        to override this and save any necessary fields from the `context`.

        Args:
          context: A `base_plugin.TBContext` object.
        """
        pass

    @abstractmethod
    def get_plugin_apps(self):
        """Returns a set of WSGI applications that the plugin implements.

        Each application gets registered with the tensorboard app and is served
        under a prefix path that includes the name of the plugin.

        Returns:
          A dict mapping route paths to WSGI applications. Each route path
          should include a leading slash.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_active(self):
        """Determines whether this plugin is active.

        A plugin may not be active for instance if it lacks relevant data. If a
        plugin is inactive, the frontend may avoid issuing requests to its routes.

        Returns:
          A boolean value. Whether this plugin is active.
        """
        raise NotImplementedError()

    def frontend_metadata(self):
        """Defines how the plugin will be displayed on the frontend.

        The base implementation returns a default value. Subclasses
        should override this and specify either an `es_module_path` or
        (for legacy plugins) an `element_name`, and are encouraged to
        set any other relevant attributes.
        """
        return FrontendMetadata()

    def data_plugin_names(self):
        """Experimental. Lists plugins whose summary data this plugin reads.

        Returns:
          A collection of strings representing plugin names (as read
          from `SummaryMetadata.plugin_data.plugin_name`) from which
          this plugin may read data. Defaults to `(self.plugin_name,)`.
        """
        return (self.plugin_name,)