import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
class _Box(ContainerWidget):
    TOP, CENTER, BOTTOM = range(3)

    def __init__(self, gravity=CENTER):
        ContainerWidget.__init__(self)
        self.gravity = gravity

    def sizeHint(self):
        height = 0
        width = 0
        for ch in self.children:
            hint = ch.sizeHint()
            if hint is None:
                hint = (None, None)
            if self.variableDimension == 0:
                if hint[0] is None:
                    width = None
                elif width is not None:
                    width += hint[0]
                if hint[1] is None:
                    height = None
                elif height is not None:
                    height = max(height, hint[1])
            else:
                if hint[0] is None:
                    width = None
                elif width is not None:
                    width = max(width, hint[0])
                if hint[1] is None:
                    height = None
                elif height is not None:
                    height += hint[1]
        return (width, height)

    def render(self, width, height, terminal):
        if not self.children:
            return
        greedy = 0
        wants = []
        for ch in self.children:
            hint = ch.sizeHint()
            if hint is None:
                hint = (None, None)
            if hint[self.variableDimension] is None:
                greedy += 1
            wants.append(hint[self.variableDimension])
        length = (width, height)[self.variableDimension]
        totalWant = sum((w for w in wants if w is not None))
        if greedy:
            leftForGreedy = int((length - totalWant) / greedy)
        widthOffset = heightOffset = 0
        for want, ch in zip(wants, self.children):
            if want is None:
                want = leftForGreedy
            subWidth, subHeight = (width, height)
            if self.variableDimension == 0:
                subWidth = want
            else:
                subHeight = want
            wrap = BoundedTerminalWrapper(terminal, subWidth, subHeight, widthOffset, heightOffset)
            ch.draw(subWidth, subHeight, wrap)
            if self.variableDimension == 0:
                widthOffset += want
            else:
                heightOffset += want