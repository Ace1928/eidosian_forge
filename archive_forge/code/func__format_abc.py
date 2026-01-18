from reportlab.rl_config import register_reset
def _format_abc(num):
    """Lowercase.  Wraps around at 26."""
    n = (num - 1) % 26
    return chr(n + 97)