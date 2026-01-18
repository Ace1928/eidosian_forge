class Primers:
    """A primer set designed by Primer3.

    Members:

        - size - length of product, note you can use len(primer) as an
          alternative to primer.size

        - forward_seq
        - forward_start
        - forward_length
        - forward_tm
        - forward_gc

        - reverse_seq
        - reverse_start
        - reverse_length
        - reverse_tm
        - reverse_gc

        - internal_seq
        - internal_start
        - internal_length
        - internal_tm
        - internal_gc

    """

    def __init__(self):
        """Initialize the class."""
        self.size = 0
        self.forward_seq = ''
        self.forward_start = 0
        self.forward_length = 0
        self.forward_tm = 0.0
        self.forward_gc = 0.0
        self.reverse_seq = ''
        self.reverse_start = 0
        self.reverse_length = 0
        self.reverse_tm = 0.0
        self.reverse_gc = 0.0
        self.internal_seq = ''
        self.internal_start = 0
        self.internal_length = 0
        self.internal_tm = 0.0
        self.internal_gc = 0.0

    def __len__(self):
        """Length of the primer product (i.e. product size)."""
        return self.size