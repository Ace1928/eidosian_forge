class PasslibHashWarning(PasslibWarning):
    """Warning issued when non-fatal issue is found with parameters
    or hash string passed to a passlib hash class.

    This occurs primarily in one of two cases:

    * A rounds value or other setting was explicitly provided which
      exceeded the handler's limits (and has been clamped
      by the :ref:`relaxed<relaxed-keyword>` flag).

    * A malformed hash string was encountered which (while parsable)
      should be re-encoded.

    .. versionadded:: 1.6
    """