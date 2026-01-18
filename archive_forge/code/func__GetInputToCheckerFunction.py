def _GetInputToCheckerFunction(self, flag_values):
    """Given flag values, construct the input to be given to checker.

    Args:
      flag_values: gflags.FlagValues
    Returns:
      dictionary, with keys() being self.lag_names, and value for each key
        being the value of the corresponding flag (string, boolean, etc).
    """
    return dict(([key, flag_values[key].value] for key in self.flag_names))