def CollectDescriptorNames(tree, names, level=0, maxDepth=100000000.0):
    if level < maxDepth:
        if not tree.GetTerminal():
            names[tree.GetLabel()] = tree.GetName()
            for child in tree.GetChildren():
                CollectDescriptorNames(child, names, level + 1, maxDepth)
    return names