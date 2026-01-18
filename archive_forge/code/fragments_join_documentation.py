from .state_inline import StateInline

    Clean up tokens after emphasis and strikethrough postprocessing:
    merge adjacent text nodes into one and re-calculate all token levels

    This is necessary because initially emphasis delimiter markers (``*, _, ~``)
    are treated as their own separate text tokens. Then emphasis rule either
    leaves them as text (needed to merge with adjacent text) or turns them
    into opening/closing tags (which messes up levels inside).
    