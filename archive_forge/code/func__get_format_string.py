def _get_format_string(self):
    if self.detail is None:
        self.detail = self._get_detail()
    return super()._get_format_string()