from typing import Set
import torch.nn as nn
[note: Dynamo treats FSDP wrapped modules as UnspecializedNNModule]

            Dynamo doesn't get to see this instance (FullyShardedDataParallel) during tracing, since
            it skips tracing all the torch.distributed.fsdp code.
                - Why? Running the FSDP code eagerly avoids lots of issues trying to trace complex hooks, and also
                gets us graph-breaks on FSDP module boundaries which we want anyway for comm ops.
                - However, we _also_ want dynamo to treat the wrapped module inside FSDP 'unspecially' (*),
                and we need a way to indicate to dynamo which modules are wrapped by FSDP.

            (*) UnspecializedNNModules in dynamo are traced-through without any assumptions, and with thorough
            guards.  NNModules otherwise are 'specialized', meaning there is less overhead due to assuming
            their code is well-behaved.

            One particular issue with specialized NNModules for FSDP is that the
            views created for orig_params are captured into the compiled graph on the first iteration, and while
            they are always going to point to the correct flatparameter and give correct results, their order
            of creation influences the order of backward execution, preventing overlap of comm and computation
            during backward.  We need to _use_ the new parameter views created on each forward iteration, in
            order for backward to interleave hooks with compute per layer.  UnspecializedNNModule lets us achieve
            this by capturing the module code more 'functionally' and passing parameters in as inputs each time.
            