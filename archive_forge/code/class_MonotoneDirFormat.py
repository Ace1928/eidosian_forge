from ... import version_info  # noqa: F401
from ... import controldir, errors
class MonotoneDirFormat(controldir.ControlDirFormat):
    """Monotone directory format."""

    def get_converter(self):
        raise NotImplementedError(self.get_converter)

    def get_format_description(self):
        return 'Monotone control directory'

    def initialize_on_transport(self, transport):
        raise errors.UninitializableFormat(self)

    def is_supported(self):
        return False

    def supports_transport(self, transport):
        return False

    def check_support_status(self, allow_unsupported, recommend_upgrade=True, basedir=None):
        raise MonotoneUnsupportedError(format=self)

    def open(self, transport):
        MonotoneProber().probe_transport(transport)
        raise NotImplementedError(self.open)