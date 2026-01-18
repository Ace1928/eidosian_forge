def _get_relative_atomic_masses():
    for mass in tuple((element[2] for element in _elements)):
        yield (float(mass[1:-1]) if str(mass).startswith('[') else float(mass))