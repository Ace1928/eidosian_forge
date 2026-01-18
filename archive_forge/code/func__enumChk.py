from functools import partial
def _enumChk(name, value, allowed=()):
    if value not in allowed:
        raise ValueError(f'invalid value {value!r} for rl_config.{name}\nneed one of {allowed}')