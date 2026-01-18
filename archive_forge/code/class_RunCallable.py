import greenlet
class RunCallable:

    def __del__(self):
        results.append(('RunCallable', '__del__'))
        main.switch('from RunCallable')