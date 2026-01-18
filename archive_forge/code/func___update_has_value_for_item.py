def __update_has_value_for_item(self, item_has_value):
    if item_has_value:
        if self.has_value() and self._extra_parameter_errors:
            self._error('got multiple values for a single choice parameter')
        self._has_value = True