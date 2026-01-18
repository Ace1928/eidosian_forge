from abc import ABCMeta
from abc import abstractmethod
class TBContext:
    """Magic container of information passed from TensorBoard core to plugins.

    A TBContext instance is passed to the constructor of a TBPlugin class. Plugins
    are strongly encouraged to assume that any of these fields can be None. In
    cases when a field is considered mandatory by a plugin, it can either crash
    with ValueError, or silently choose to disable itself by returning False from
    its is_active method.

    All fields in this object are thread safe.
    """

    def __init__(self, *, assets_zip_provider=None, data_provider=None, flags=None, logdir=None, multiplexer=None, plugin_name_to_instance=None, sampling_hints=None, window_title=None):
        """Instantiates magic container.

        The argument list is sorted and may be extended in the future; therefore,
        callers must pass only named arguments to this constructor.

        Args:
          assets_zip_provider: A function that returns a newly opened file handle
              for a zip file containing all static assets. The file names inside the
              zip file are considered absolute paths on the web server. The file
              handle this function returns must be closed. It is assumed that you
              will pass this file handle to zipfile.ZipFile. This zip file should
              also have been created by the tensorboard_zip_file build rule.
          data_provider: Instance of `tensorboard.data.provider.DataProvider`. May
            be `None` if `flags.generic_data` is set to `"false"`.
          flags: An object of the runtime flags provided to TensorBoard to their
              values.
          logdir: The string logging directory TensorBoard was started with.
          multiplexer: An EventMultiplexer with underlying TB data. Plugins should
              copy this data over to the database when the db fields are set.
          plugin_name_to_instance: A mapping between plugin name to instance.
              Plugins may use this property to access other plugins. The context
              object is passed to plugins during their construction, so a given
              plugin may be absent from this mapping until it is registered. Plugin
              logic should handle cases in which a plugin is absent from this
              mapping, lest a KeyError is raised.
          sampling_hints: Map from plugin name to `int` or `NoneType`, where
              the value represents the user-specified downsampling limit as
              given to the `--samples_per_plugin` flag, or `None` if none was
              explicitly given for this plugin.
          window_title: A string specifying the window title.
        """
        self.assets_zip_provider = assets_zip_provider
        self.data_provider = data_provider
        self.flags = flags
        self.logdir = logdir
        self.multiplexer = multiplexer
        self.plugin_name_to_instance = plugin_name_to_instance
        self.sampling_hints = sampling_hints
        self.window_title = window_title