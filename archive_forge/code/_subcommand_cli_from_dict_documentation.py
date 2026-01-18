from typing import Any, Callable, Dict, Optional, Sequence, TypeVar, Union, overload
from typing_extensions import Annotated
from .._cli import cli
from ..conf import subcommand
Generate a subcommand CLI from a dictionary of functions.

    For an input like:

    ```python
    tyro.extras.subcommand_cli_from_dict(
        {
            "checkout": checkout,
            "commit": commit,
        }
    )
    ```

    This is internally accomplished by generating and calling:

    ```python
    from typing import Annotated, Any, Union
    import tyro

    tyro.cli(
        Union[
            Annotated[
                Any,
                tyro.conf.subcommand(name="checkout", constructor=checkout),
            ],
            Annotated[
                Any,
                tyro.conf.subcommand(name="commit", constructor=commit),
            ],
        ]
    )
    ```

    Args:
        subcommands: Dictionary that maps the subcommand name to function to call.
        prog: The name of the program printed in helptext. Mirrors argument from
            `argparse.ArgumentParser()`.
        description: Description text for the parser, displayed when the --help flag is
            passed in. If not specified, `f`'s docstring is used. Mirrors argument from
            `argparse.ArgumentParser()`.
        args: If set, parse arguments from a sequence of strings instead of the
            commandline. Mirrors argument from `argparse.ArgumentParser.parse_args()`.
        use_underscores: If True, use underscores as a word delimeter instead of hyphens.
            This primarily impacts helptext; underscores and hyphens are treated equivalently
            when parsing happens. We default helptext to hyphens to follow the GNU style guide.
            https://www.gnu.org/software/libc/manual/html_node/Argument-Syntax.html
    