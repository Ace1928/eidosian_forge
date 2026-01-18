from typing import Dict, Optional, Tuple
class Mailmap:
    """Class for accessing a mailmap file."""

    def __init__(self, map=None) -> None:
        self._table: Dict[Tuple[Optional[str], str], Tuple[str, str]] = {}
        if map:
            for canonical_identity, from_identity in map:
                self.add_entry(canonical_identity, from_identity)

    def add_entry(self, canonical_identity, from_identity=None):
        """Add an entry to the mail mail.

        Any of the fields can be None, but at least one of them needs to be
        set.

        Args:
          canonical_identity: The canonical identity (tuple)
          from_identity: The from identity (tuple)
        """
        if from_identity is None:
            from_name, from_email = (None, None)
        else:
            from_name, from_email = from_identity
        canonical_name, canonical_email = canonical_identity
        if from_name is None and from_email is None:
            self._table[canonical_name, None] = canonical_identity
            self._table[None, canonical_email] = canonical_identity
        else:
            self._table[from_name, from_email] = canonical_identity

    def lookup(self, identity):
        """Lookup an identity in this mailmail."""
        if not isinstance(identity, tuple):
            was_tuple = False
            identity = parse_identity(identity)
        else:
            was_tuple = True
        for query in [identity, (None, identity[1]), (identity[0], None)]:
            canonical_identity = self._table.get(query)
            if canonical_identity is not None:
                identity = (canonical_identity[0] or identity[0], canonical_identity[1] or identity[1])
                break
        if was_tuple:
            return identity
        else:
            return identity[0] + b' <' + identity[1] + b'>'

    @classmethod
    def from_path(cls, path):
        with open(path, 'rb') as f:
            return cls(read_mailmap(f))