from ... import version_info  # noqa: F401
from ... import controldir, errors
class FossilDirFormat(controldir.ControlDirFormat):
    """Fossil directory format."""

    def get_converter(self):
        raise NotImplementedError(self.get_converter)

    def get_format_description(self):
        return 'Fossil control directory'

    def initialize_on_transport(self, transport):
        raise errors.UninitializableFormat(format=self)

    def is_supported(self):
        return False

    def supports_transport(self, transport):
        return False

    def check_support_status(self, allow_unsupported, recommend_upgrade=True, basedir=None):
        raise FossilUnsupportedError(format=self)

    def open(self, transport):
        RemoteFossilProber().probe_transport(transport)
        raise NotImplementedError(self.open)