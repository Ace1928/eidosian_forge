from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from .. import frames
from ..typing import ExtensionName, ExtensionParameter
class ClientExtensionFactory:
    """
    Base class for client-side extension factories.

    """
    name: ExtensionName
    'Extension identifier.'

    def get_request_params(self) -> List[ExtensionParameter]:
        """
        Build parameters to send to the server for this extension.

        Returns:
            List[ExtensionParameter]: Parameters to send to the server.

        """
        raise NotImplementedError

    def process_response_params(self, params: Sequence[ExtensionParameter], accepted_extensions: Sequence[Extension]) -> Extension:
        """
        Process parameters received from the server.

        Args:
            params (Sequence[ExtensionParameter]): parameters received from
                the server for this extension.
            accepted_extensions (Sequence[Extension]): list of previously
                accepted extensions.

        Returns:
            Extension: An extension instance.

        Raises:
            NegotiationError: if parameters aren't acceptable.

        """
        raise NotImplementedError