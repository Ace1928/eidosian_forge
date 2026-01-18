import sys
from proto.marshal import Marshal
Return the package and marshal to use.

    Args:
        name (str): The name of the new class, as sent to ``type.__new__``.
        attrs (Mapping[str, Any]): The attrs for a new class, as sent
            to ``type.__new__``

    Returns:
        Tuple[str, ~.Marshal]:
            - The proto package, if any (empty string otherwise).
            - The marshal object to use.
    