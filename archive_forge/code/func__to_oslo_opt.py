import itertools
import os
from keystoneauth1.loading import _utils
def _to_oslo_opt(self):
    cfg = _utils.get_oslo_config()
    deprecated_opts = [cfg.DeprecatedOpt(o.name) for o in self.deprecated]
    return cfg.Opt(name=self.name, type=self.type, help=self.help, secret=self.secret, required=self.required, dest=self.dest, deprecated_opts=deprecated_opts, metavar=self.metavar)