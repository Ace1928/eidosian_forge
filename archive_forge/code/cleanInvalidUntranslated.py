import sys
import os.path
from os import path

fileName = sys.argv[1]

if path.exists(fileName):

    itemPrefix = "#. "
    translationVariableName = "msgstr"
    emptyItemPrefix = "\nmsgid \"\"" + "\n" + translationVariableName #most of empty ones look like this, but there are some msgid that have multiple lines, and the first line begins like an empty one, those are trickier

    file = open(fileName, "r")
    text = file.read()
    file.close()

    items = text.split(itemPrefix)
    header = items.pop(0)

    notEmptyItems = items.copy() 

    counter = 0
    emptyFilesFoundCounter = 0
    for item in items:
        if item.find(emptyItemPrefix) != -1:
            notEmptyItems.pop(counter)
            counter -= 1
            emptyFilesFoundCounter += 1
        
        counter += 1

    result = header
    for item in notEmptyItems:
        result += itemPrefix + item

    #print(result)
    file = open(fileName, "w")
    file.seek(0) # places the pointer on the line zero
    file.write(result)
    file.truncate() # deletes the rest
    file.close()

    inform = "\nUntranslated strings found: " + str(counter)
    print(inform)
        