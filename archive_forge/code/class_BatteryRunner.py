class BatteryRunner:
    """Class to run set of checks"""

    def __init__(self, checks):
        """Initialize instance from sequence of `checks`

        Parameters
        ----------
        checks : sequence
           sequence of checks, where checks are callables matching
           signature ``obj, rep = chk(obj, fix=False)``.  Checks are run
           in the order they are passed.

        Examples
        --------
        >>> def chk(obj, fix=False): # minimal check
        ...     return obj, Report()
        >>> btrun = BatteryRunner((chk,))
        """
        self._checks = checks

    def check_only(self, obj):
        """Run checks on `obj` returning reports

        Parameters
        ----------
        obj : anything
           object on which to run checks

        Returns
        -------
        reports : sequence
           sequence of report objects reporting on result of running
           checks (without fixes) on `obj`
        """
        reports = []
        for check in self._checks:
            obj, rep = check(obj, False)
            reports.append(rep)
        return reports

    def check_fix(self, obj):
        """Run checks, with fixes, on `obj` returning `obj`, reports

        Parameters
        ----------
        obj : anything
           object on which to run checks, fixes

        Returns
        -------
        obj : anything
           possibly modified or replaced `obj`, after fixes
        reports : sequence
           sequence of reports on checks, fixes
        """
        reports = []
        for check in self._checks:
            obj, report = check(obj, True)
            reports.append(report)
        return (obj, reports)

    def __len__(self):
        return len(self._checks)