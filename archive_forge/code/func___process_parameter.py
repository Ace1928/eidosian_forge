def __process_parameter(self, param_name, param_type, ancestry=None):
    """Collect values for a given web service operation input parameter."""
    assert self.active()
    param_optional = param_type.optional()
    has_argument, value = self.__get_param_value(param_name)
    if has_argument:
        self.__params_with_arguments.add(param_name)
    self.__update_context(ancestry)
    self.__stack[-1].process_parameter(param_optional, value is not None)
    self.__external_param_processor(param_name, param_type, self.__in_choice_context(), value)