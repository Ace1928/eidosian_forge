from keystoneauth1 import exceptions
from keystoneauth1.loading import base
from keystoneauth1.loading import opts
class BaseV3Loader(BaseIdentityLoader):
    """Base Option handling for identity plugins.

    This class defines options and handling that should be common to the V3
    identity API. It provides the options expected by the
    :py:class:`keystoneauth1.identity.v3.Auth` class.
    """

    def get_options(self):
        options = super(BaseV3Loader, self).get_options()
        options.extend([opts.Opt('system-scope', help='Scope for system operations'), opts.Opt('domain-id', help='Domain ID to scope to'), opts.Opt('domain-name', help='Domain name to scope to'), opts.Opt('project-id', help='Project ID to scope to'), opts.Opt('project-name', help='Project name to scope to'), opts.Opt('project-domain-id', help='Domain ID containing project'), opts.Opt('project-domain-name', help='Domain name containing project'), opts.Opt('trust-id', help='ID of the trust to use as a trustee use')])
        return options

    def load_from_options(self, **kwargs):
        if kwargs.get('project_name') and (not (kwargs.get('project_domain_name') or kwargs.get('project_domain_id'))):
            m = 'You have provided a project_name. In the V3 identity API a project_name is only unique within a domain so you must also provide either a project_domain_id or project_domain_name.'
            raise exceptions.OptionError(m)
        return super(BaseV3Loader, self).load_from_options(**kwargs)