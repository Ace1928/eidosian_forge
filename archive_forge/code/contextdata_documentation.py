from OpenGL import platform
import weakref
Cleanup all held pointer objects for the given context
    
    Warning: this is dangerous, as if you call it before a context 
    is destroyed you may release memory held by the context and cause
    a protection fault when the GL goes to render the scene!
    
    Normally you will want to get the context ID explicitly and then 
    register cleanupContext as a weakref callback to your GUI library 
    Context object with the (now invalid) context ID as parameter.
    