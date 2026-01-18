import REPL.REPLCompletions
import REPL.REPLCompletions
import REPL.REPLCompletions

#!/usr/bin/env julia

import REPL.REPLCompletions
res = String["true", "false"]
for compl in filter!(x -> isa(x, REPLCompletions.ModuleCompletion) && (x.parent === Base || x.parent === Core),
                    REPLCompletions.completions("", 0)[1])
    try
        v = eval(Symbol(compl.mod))
        if !(v isa Function || v isa Type || v isa TypeVar || v isa Module || v isa Colon)
            push!(res, compl.mod)
        end
    catch e
    end
end
sort!(unique!(res))
foreach(x -> println("'", x, "',"), res)
