def activate_writer_version(name, ver):
    """DEBUGGING TOOL to switch the "default" writer implementation"""
    doc = WriterFactory.doc(name)
    WriterFactory.unregister(name)
    WriterFactory.register(name, doc)(WriterFactory.get_class(f'{name}_v{ver}'))