import traitlets.config
from traitlets import Unicode, Bool, Dict, List, Int, Enum, Type
class VoilaConfiguration(traitlets.config.Configurable):
    """Common configuration options between the server extension and the application."""
    allow_template_override = Enum(['YES', 'NOTEBOOK', 'NO'], 'YES', help='\n    Allow overriding the template (YES), or not (NO), or only from the notebook metadata.\n    ', config=True)
    allow_theme_override = Enum(['YES', 'NOTEBOOK', 'NO'], 'YES', help='\n    Allow overriding the theme (YES), or not (NO), or only from the notebook metadata.\n    ', config=True)
    template = Unicode('lab', config=True, allow_none=True, help='template name to be used by voila.')
    resources = Dict(allow_none=True, config=True, help='\n        extra resources used by templates;\n        example use with --template=reveal\n        --VoilaConfiguration.resources="{\'reveal\': {\'transition\': \'fade\', \'scroll\': True}}"\n        ')
    theme = Unicode('light', config=True)
    show_margins = Bool(False, config=True, help='Show left and right margins for the "lab" template, this gives a "classic" template look')
    strip_sources = Bool(True, config=True, help='Strip sources from rendered html')
    enable_nbextensions = Bool(False, config=True, help='Set to True for Voilà to load notebook extensions')
    nbextensions_path = Unicode('', config=True, help='Set to override default path provided by Jupyter Server')
    file_whitelist = List(Unicode(), ['.*\\.(png|jpg|gif|svg)'], config=True, help='\n    List of regular expressions for controlling which static files are served.\n    All files that are served should at least match 1 whitelist rule, and no blacklist rule\n    Example: --VoilaConfiguration.file_whitelist="[\'.*\\.(png|jpg|gif|svg)\', \'public.*\']"\n    ')
    file_blacklist = List(Unicode(), ['.*\\.(ipynb|py)'], config=True, help='\n    List of regular expressions for controlling which static files are forbidden to be served.\n    All files that are served should at least match 1 whitelist rule, and no blacklist rule\n    Example:\n    --VoilaConfiguration.file_whitelist="[\'.*\']" # all files\n    --VoilaConfiguration.file_blacklist="[\'private.*\', \'.*\\.(ipynb)\']" # except files in the private dir and notebook files\n    ')
    language_kernel_mapping = Dict({}, config=True, help='Mapping of language name to kernel name\n        Example mapping python to use xeus-python, and C++11 to use xeus-cling:\n        --VoilaConfiguration.extension_language_mapping=\'{"python": "xpython", "C++11": "xcpp11"}\'\n        ')
    extension_language_mapping = Dict({}, config=True, help='Mapping of file extension to kernel language\n        Example mapping .py files to a python language kernel, and .cpp to a C++11 language kernel:\n        --VoilaConfiguration.extension_language_mapping=\'{".py": "python", ".cpp": "C++11"}\'\n        ')
    http_keep_alive_timeout = Int(10, config=True, help="\n    When a cell takes a long time to execute, the http connection can timeout (possibly because of a proxy).\n    Voila sends a 'heartbeat' message after the timeout is passed to keep the http connection alive.\n    ")
    show_tracebacks = Bool(False, config=True, help='Whether to send tracebacks to clients on exceptions.')
    multi_kernel_manager_class = Type(config=True, default_value='jupyter_server.services.kernels.kernelmanager.AsyncMappingKernelManager', klass='jupyter_client.multikernelmanager.MultiKernelManager', help='The kernel manager class. This is useful to specify a different kernel manager,\n        for example a kernel manager with support for pooling.\n        ')
    http_header_envs = List(Unicode(), [], help='\n    List of HTTP Headers that should be passed as env vars to the kernel.\n    Example: --VoilaConfiguration.http_header_envs="[\'X-CDSDASHBOARDS-JH-USER\']"\n    ').tag(config=True)
    preheat_kernel = Bool(False, config=True, help='Flag to enable or disable pre-heat kernel option.\n        ')
    default_pool_size = Int(1, config=True, help='Size of pre-heated kernel pool for each notebook. Zero or negative number means disabled.\n        ')