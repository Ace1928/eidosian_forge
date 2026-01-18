class AnimationFrame:
    """A single frame of an animation."""
    __slots__ = ('image', 'duration')

    def __init__(self, image, duration):
        """Create an animation frame from an image.

        :Parameters:
            `image` : `~pyglet.image.AbstractImage`
                The image of this frame.
            `duration` : float
                Number of seconds to display the frame, or ``None`` if it is
                the last frame in the animation.

        """
        self.image = image
        self.duration = duration

    def __repr__(self):
        return 'AnimationFrame({0}, duration={1})'.format(self.image, self.duration)