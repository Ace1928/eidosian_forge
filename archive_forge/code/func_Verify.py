def Verify(self, flag_values):
    """Verify that constraint is satisfied.

    flags library calls this method to verify Validator's constraint.
    Args:
      flag_values: gflags.FlagValues, containing all flags
    Raises:
      Error: if constraint is not satisfied.
    """
    param = self._GetInputToCheckerFunction(flag_values)
    if not self.checker(param):
        raise Error(self.message)