Write report to `stream`

        Parameters
        ----------
        stream : file-like
           implementing ``write`` method
        error_level : int, optional
           level at which to raise error for problem detected in
           ``self``
        log_level : int, optional
           Such that if `log_level` is >= ``self.problem_level`` we
           write the report to `stream`, otherwise we write nothing.
        