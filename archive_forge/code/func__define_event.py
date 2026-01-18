def _define_event(callback_function):
    available_events[callback_function.__name__] = callback_function
    return callback_function