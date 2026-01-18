from __future__ import annotations
@classmethod
def _during(cls, step: str, var: str='') -> PlotSpecError:
    """
        Initialize the class to report the failure of a specific operation.
        """
    message = []
    if var:
        message.append(f'{step} failed for the `{var}` variable.')
    else:
        message.append(f'{step} failed.')
    message.append('See the traceback above for more information.')
    return cls(' '.join(message))