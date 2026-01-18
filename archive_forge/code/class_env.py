import os
from argparse import Action
class env(Action):
    """
    Get argument values from ``PET_{dest}`` before defaultingto the given ``default`` value.

    For flags (e.g. ``--standalone``)
    use ``check_env`` instead.

    .. note:: when multiple option strings are specified, ``dest`` is
              the longest option string (e.g. for ``"-f", "--foo"``
              the env var to set is ``PET_FOO`` not ``PET_F``)

    Example:
    ::

     parser.add_argument("-f", "--foo", action=env, default="bar")

     ./program                                      -> args.foo="bar"
     ./program -f baz                               -> args.foo="baz"
     ./program --foo baz                            -> args.foo="baz"
     PET_FOO="env_bar" ./program -f baz    -> args.foo="baz"
     PET_FOO="env_bar" ./program --foo baz -> args.foo="baz"
     PET_FOO="env_bar" ./program           -> args.foo="env_bar"

     parser.add_argument("-f", "--foo", action=env, required=True)

     ./program                                      -> fails
     ./program -f baz                               -> args.foo="baz"
     PET_FOO="env_bar" ./program           -> args.foo="env_bar"
     PET_FOO="env_bar" ./program -f baz    -> args.foo="baz"
    """

    def __init__(self, dest, default=None, required=False, **kwargs) -> None:
        env_name = f'PET_{dest.upper()}'
        default = os.environ.get(env_name, default)
        if default:
            required = False
        super().__init__(dest=dest, default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)