def PrintFlagsWithValues(self, flag_values):
    prefix = 'flags '
    flags_with_values = []
    for key in self.flag_names:
        flags_with_values.append('%s=%s' % (key, flag_values[key].value))
    return prefix + ', '.join(flags_with_values)