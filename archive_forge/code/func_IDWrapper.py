def IDWrapper(freer):
    """
    Classes create with IDWrapper return an ID and then free it upon exit of the
    context.
    """

    class Wrapper:

        def __init__(self, conn):
            self.conn = conn
            self.id = None

        def __enter__(self):
            self.id = self.conn.generate_id()
            return self.id

        def __exit__(self, exception_type, exception_value, traceback):
            getattr(self.conn.core, freer)(self.id)
    return Wrapper