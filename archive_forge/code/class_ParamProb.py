import abc
class ParamProb:
    """An abstract base class for parameterized problems.

    Parameterized problems are produced during the first canonicalization
    and allow canonicalization to be short-circuited for future solves.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def is_mixed_integer(self) -> bool:
        """Is the problem mixed-integer?"""
        raise NotImplementedError()

    @abc.abstractproperty
    def apply_parameters(self, id_to_param_value=None, zero_offset: bool=False, keep_zeros: bool=False):
        """Returns A, b after applying parameters (and reshaping).

        Args:
          id_to_param_value: (optional) dict mapping parameter ids to values
          zero_offset: (optional) if True, zero out the constant offset in the
                       parameter vector
          keep_zeros: (optional) if True, store explicit zeros in A where
                        parameters are affected
        """
        raise NotImplementedError()