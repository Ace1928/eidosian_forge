from . import errors, registry, urlutils
class URLParametrizedIntegerBugTracker(IntegerBugTracker, URLParametrizedBugTracker):
    """A type of bug tracker that  only allows integer bug IDs.

    This can be found on a variety of different sites, and thus needs to have
    the base URL configured.

    Looks for a config setting in the form '<type_name>_<abbreviation>_url'.
    `type_name` is the name of the type of tracker (e.g. 'bugzilla' or 'trac')
    and `abbreviation` is a short name for the particular instance (e.g.
    'squid' or 'apache').
    """