from itertools import chain
from graphql import Undefined
from .dynamic import Dynamic
from .mountedtype import MountedType
from .structures import NonNull
from .utils import get_type

    Makes an Argument available on a Field in the GraphQL schema.

    Arguments will be parsed and provided to resolver methods for fields as keyword arguments.

    All ``arg`` and ``**extra_args`` for a ``graphene.Field`` are implicitly mounted as Argument
    using the below parameters.

    .. code:: python

        from graphene import String, Boolean, Argument

        age = String(
            # Boolean implicitly mounted as Argument
            dog_years=Boolean(description="convert to dog years"),
            # Boolean explicitly mounted as Argument
            decades=Argument(Boolean, default_value=False),
        )

    args:
        type (class for a graphene.UnmountedType): must be a class (not an instance) of an
            unmounted graphene type (ex. scalar or object) which is used for the type of this
            argument in the GraphQL schema.
        required (optional, bool): indicates this argument as not null in the graphql schema. Same behavior
            as graphene.NonNull. Default False.
        name (optional, str): the name of the GraphQL argument. Defaults to parameter name.
        description (optional, str): the description of the GraphQL argument in the schema.
        default_value (optional, Any): The value to be provided if the user does not set this argument in
            the operation.
        deprecation_reason (optional, str): Setting this value indicates that the argument is
            depreciated and may provide instruction or reason on how for clients to proceed. Cannot be
            set if the argument is required (see spec).
    