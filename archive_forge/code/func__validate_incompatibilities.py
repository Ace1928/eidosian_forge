from Bio.Application import _Option, AbstractCommandline, _Switch
def _validate_incompatibilities(self, incompatibles):
    """Validate parameters for incompatibilities (PRIVATE).

        Used by the _validate method.
        """
    for a in incompatibles:
        if self._get_parameter(a):
            for b in incompatibles[a]:
                if self._get_parameter(b):
                    raise ValueError(f'Options {a} and {b} are incompatible.')