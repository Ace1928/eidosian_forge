class smartDecode:

    @staticmethod
    def __call__(s):
        import chardet

        def __call__(s):
            if isinstance(s, str):
                return s
            cdd = chardet.detect(s)
            return s.decode(cdd['encoding'])
        smartDecode.__class__.__call__ = staticmethod(__call__)
        return __call__(s)