import functools
import logging
def _disable_progress(e: Exception) -> None:
    """Print an exception and disable progress bars for this session"""
    global _SHOW_PROGRESS
    if _SHOW_PROGRESS:
        _SHOW_PROGRESS = False
        logging.getLogger('cmdstanpy').error('Error in progress bar initialization:\n\t%s\nDisabling progress bars for this session', str(e))