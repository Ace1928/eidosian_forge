from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
def _mouseMoveDrag(moveOrDrag, x, y, xOffset, yOffset, duration, tween=linear, button=None):
    """Handles the actual move or drag event, since different platforms
    implement them differently.

    On Windows & Linux, a drag is a normal mouse move while a mouse button is
    held down. On OS X, a distinct "drag" event must be used instead.

    The code for moving and dragging the mouse is similar, so this function
    handles both. Users should call the moveTo() or dragTo() functions instead
    of calling _mouseMoveDrag().

    Args:
      moveOrDrag (str): Either 'move' or 'drag', for the type of action this is.
      x (int, float, None, optional): How far left (for negative values) or
        right (for positive values) to move the cursor. 0 by default.
      y (int, float, None, optional): How far up (for negative values) or
        down (for positive values) to move the cursor. 0 by default.
      xOffset (int, float, None, optional): How far left (for negative values) or
        right (for positive values) to move the cursor. 0 by default.
      yOffset (int, float, None, optional): How far up (for negative values) or
        down (for positive values) to move the cursor. 0 by default.
      duration (float, optional): The amount of time it takes to move the mouse
        cursor to the new xy coordinates. If 0, then the mouse cursor is moved
        instantaneously. 0.0 by default.
      tween (func, optional): The tweening function used if the duration is not
        0. A linear tween is used by default.
      button (str, int, optional): The mouse button released. TODO

    Returns:
      None
    """
    assert moveOrDrag in ('move', 'drag'), "moveOrDrag must be in ('move', 'drag'), not %s" % moveOrDrag
    if sys.platform != 'darwin':
        moveOrDrag = 'move'
    xOffset = int(xOffset) if xOffset is not None else 0
    yOffset = int(yOffset) if yOffset is not None else 0
    if x is None and y is None and (xOffset == 0) and (yOffset == 0):
        return
    startx, starty = position()
    x = int(x) if x is not None else startx
    y = int(y) if y is not None else starty
    x += xOffset
    y += yOffset
    width, height = size()
    steps = [(x, y)]
    if duration > MINIMUM_DURATION:
        num_steps = max(width, height)
        sleep_amount = duration / num_steps
        if sleep_amount < MINIMUM_SLEEP:
            num_steps = int(duration / MINIMUM_SLEEP)
            sleep_amount = duration / num_steps
        steps = [getPointOnLine(startx, starty, x, y, tween(n / num_steps)) for n in range(num_steps)]
        steps.append((x, y))
    for tweenX, tweenY in steps:
        if len(steps) > 1:
            time.sleep(sleep_amount)
        tweenX = int(round(tweenX))
        tweenY = int(round(tweenY))
        if (tweenX, tweenY) not in FAILSAFE_POINTS:
            failSafeCheck()
        if moveOrDrag == 'move':
            platformModule._moveTo(tweenX, tweenY)
        elif moveOrDrag == 'drag':
            platformModule._dragTo(tweenX, tweenY, button)
        else:
            raise NotImplementedError('Unknown value of moveOrDrag: {0}'.format(moveOrDrag))
    if (tweenX, tweenY) not in FAILSAFE_POINTS:
        failSafeCheck()