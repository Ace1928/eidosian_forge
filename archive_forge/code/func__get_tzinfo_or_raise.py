def _get_tzinfo_or_raise(tzenv: str):
    tzinfo = _get_tzinfo(tzenv)
    if tzinfo is None:
        raise LookupError(f'Can not find timezone {tzenv}. \nTimezone names are generally in the form `Continent/City`.')
    return tzinfo